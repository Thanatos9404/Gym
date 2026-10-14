# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the task images a Dockerfile overlay cannot express (image mode three of three).

Prepare decides one of three modes per distinct Dockerfile. Pull, for a base-image-only file.
Overlay, for a pullable ``FROM`` plus ``RUN``, ``WORKDIR``, ``ENV`` and ``USER`` lines the
server applies at seed (see ``nemo_gym.tasks.harbor.dockerfile``). Build, for everything else
(``COPY``, ``ADD``, heredocs, BuildKit flags, multi-stage files): the task's ``environment/``
folder is the build context, built once with buildah, podman or docker, pushed to a registry
when one is named, and recorded in ``built-images.json`` next to the task folders, keyed by
Harbor's content hash of the build context. The loader then runs the task in pull mode from the
recorded image, with ``WORKDIR``, ``ENV`` and ``USER`` read from the image's recorded configuration
in ``compose-images.json`` like any other pulled image.

The hash and the image name are Harbor's (``nemo_gym.tasks.harbor.harbor_compat``, vendored from
Harbor v0.22.0): the hash is Harbor's environment id of the folder and the image is ``hb__<hash>``,
under the repository's registry path when one is named, so identical contexts get the same name
in Harbor and in Gym.

A build is cancel-safe: nothing is recorded until the tool has built, pushed and inspected the
image, and each image is recorded as soon as it is done, so an interrupted prepare resumes
where it stopped.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.tasks.harbor.harbor_compat import sanitize_docker_image_name
from nemo_gym.tasks.harbor.image_configs import PLATFORM, RECORDED_CONFIG_KEYS
from nemo_gym.tasks.harbor.task import (
    BUILT_IMAGES_FILE,
    DOCKERFILE,
    IMAGE_CONFIGS_FILE,
    TASK_FILE,
    HarborTaskConfig,
    is_task_folder,
    read_built_images,
    read_image_configs,
)


LOGGER = logging.getLogger(__name__)

BUILD_TOOLS = ("buildah", "podman", "docker")
DEFAULT_PLATFORM = "/".join(PLATFORM)
REPOSITORY_ENV = "NEMO_GYM_BUILD_REPOSITORY"

# The seam tests replace: run one tool command and return its completed process (text mode).
Runner = Callable[[list[str]], subprocess.CompletedProcess]


class BuildError(RuntimeError):
    """A build, push or inspect command failed; nothing was recorded for the image."""


def default_storage_root() -> Path:
    """Where buildah keeps its layers: node-local when the cluster scratch disk exists, else the temp dir."""
    scratch = Path("/raid/scratch")
    if scratch.is_dir():
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "nemo-gym"
        return scratch / user / "nemo-gym-buildah"
    return Path(tempfile.gettempdir()) / "nemo-gym-buildah"


def detect_tool(repository: str | None = None) -> str | None:
    """The builder to use, or ``None`` when none is installed.

    With a repository the image is pushed, so any tool serves: buildah, then podman, then docker.
    Without one the image has to stay where a sandbox provider can start it, and only the Docker
    daemon is such a place (buildah and podman keep images in a store no provider reads), so only
    docker qualifies.
    """
    candidates = BUILD_TOOLS if repository else ("docker",)
    for tool in candidates:
        if shutil.which(tool):
            return tool
    return None


def needs_build(dockerfile_text: str, image_configs: dict[str, Any] | None = None) -> bool:
    """Whether a Dockerfile is outside what pull and overlay mode can run.

    A file whose instructions can only be resolved against its base image's configuration is
    resolved against the record in ``image_configs`` (``compose-images.json``, as prepare records
    it before planning builds). Without a record the file is classed as a build, so a context is
    never left unbuilt because the base image was unknown.
    """
    try:
        parsed = parse_dockerfile(dockerfile_text)
    except DockerfileNeedsBuild:
        return True
    try:
        resolve_dockerfile(parsed, None)
        return False
    except DockerfileNeedsBuild:
        return True
    except ImageConfigRequired:
        pass
    record = (image_configs or {}).get(parsed.image)
    if record is None:
        return True
    try:
        resolve_dockerfile(parsed, BaseImageConfig.from_record(record))
    except (DockerfileNeedsBuild, ImageConfigRequired):
        return True
    return False


@dataclass(frozen=True)
class BuildPlanEntry:
    """One distinct build context and the tasks that share it.

    ``digest`` is Harbor's environment id of the context (32 hex characters), the key of the build
    record and the body of the image name.
    """

    digest: str
    context: Path
    tasks: tuple[str, ...]


def plan_builds(folder: Path, *, select: Callable[[str], bool] | None = None) -> list[BuildPlanEntry]:
    """The distinct build contexts among the tasks under ``folder`` whose Dockerfile needs a build.

    ``folder`` is a task folder or a folder of task folders; ``select`` limits the scan to the
    tasks whose folder name it accepts. A task counts when it has ``environment/Dockerfile``,
    declares no ``[environment].docker_image`` after the dataset's overrides, and the Dockerfile
    is outside pull and overlay mode. Tasks whose ``task.toml`` does not load are left for the
    loader to report. Entries are ordered by the first task that uses them.
    """
    folder = Path(folder).resolve()
    single = is_task_folder(folder)
    children = [folder] if single else sorted(child for child in folder.iterdir() if child.is_dir())
    parent = folder.parent if single else folder
    try:
        # dataset.toml sits beside the task folders, or one level up for datasets/<name>/tasks/.
        dataset = read_dataset_config(parent)
    except ValueError:
        return []
    image_configs = read_image_configs(parent)
    entries: dict[str, list[str]] = {}
    contexts: dict[str, Path] = {}
    for child in children:
        if not is_task_folder(child) or (select is not None and not select(child.name)):
            continue
        dockerfile = child / DOCKERFILE
        if not dockerfile.is_file():
            continue
        try:
            config = HarborTaskConfig.model_validate(tomllib.loads((child / TASK_FILE).read_text()))
            config = apply_dataset_config(config, child.name, dataset)
            text = dockerfile.read_text()
        except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
            continue
        if config.environment.docker_image or not needs_build(text, image_configs):
            continue
        digest = context_digest(dockerfile.parent)
        entries.setdefault(digest, []).append(child.name)
        contexts.setdefault(digest, dockerfile.parent)
    return [BuildPlanEntry(digest, contexts[digest], tuple(tasks)) for digest, tasks in entries.items()]


@dataclass(frozen=True)
class BuildResult:
    """A built image, ready to record."""

    digest: str
    tasks: tuple[str, ...]
    tag: str
    image: str
    tool: str
    pushed: bool
    record: dict[str, Any] = field(default_factory=dict)


def _run_default(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, errors="replace")


class ImageBuilder:
    """Build one context into a named image with buildah, podman or docker, and push it when asked.

    The image is named as Harbor names it, ``hb__<hash>`` (see :meth:`tag_for`). ``repository`` is
    the registry path the name goes under (``registry/path/hb__<hash>``) and the image is pushed
    there. Without a repository the image stays local under that name in the Docker daemon, the
    only local store a sandbox provider starts images from, so buildah and podman need a
    repository. ``push`` defaults to whether a repository was named. ``platform`` (``linux/amd64``
    by default, the platform sandboxes run) is passed to the tool, and an image that comes out for
    another platform is refused before it is pushed or recorded. ``run`` is the command seam.
    """

    def __init__(
        self,
        *,
        tool: str | None = None,
        repository: str | None = None,
        push: bool | None = None,
        platform: str | None = None,
        storage_root: Path | None = None,
        run: Runner | None = None,
    ) -> None:
        self.repository = repository.rstrip("/") if repository else None
        self.tool = tool or detect_tool(self.repository)
        if self.tool is None:
            wanted = (
                ", ".join(BUILD_TOOLS) if self.repository else "docker (buildah and podman need builds.repository)"
            )
            raise BuildError(f"No image builder found: install {wanted}, or name one with builds.tool")
        if self.tool not in BUILD_TOOLS:
            raise BuildError(f"Unknown build tool {self.tool!r}; expected one of " + ", ".join(BUILD_TOOLS))
        if self.tool != "docker" and not self.repository:
            raise BuildError(
                f"{self.tool} keeps built images in a store no sandbox provider reads; set builds.repository "
                "so the image is pushed, or build with docker to keep it in the local daemon"
            )
        self.push = bool(self.repository) if push is None else push
        if self.push and not self.repository:
            raise BuildError("Pushing a built image needs builds.repository")
        self.platform = platform or DEFAULT_PLATFORM
        self.storage_root = Path(storage_root) if storage_root is not None else default_storage_root()
        self._run = run or _run_default

    # -- commands ------------------------------------------------------------------------------

    def _storage_flags(self) -> list[str]:
        if self.tool != "buildah":
            return []
        return [
            "--storage-driver",
            "vfs",
            "--root",
            str(self.storage_root / "storage"),
            "--runroot",
            str(self.storage_root / "run"),
        ]

    def tag_for(self, entry: BuildPlanEntry) -> str:
        """Harbor's name for the image, ``hb__<hash>``, under the repository when there is one.

        The hash is the entry's digest, Harbor's environment id of the context, so identical
        contexts get the same name in Harbor and in Gym.
        """
        name = sanitize_docker_image_name(f"hb__{entry.digest}")
        return f"{self.repository}/{name}" if self.repository else name

    def build_argv(self, entry: BuildPlanEntry, tag: str) -> list[str]:
        platform = ["--platform", self.platform] if self.platform else []
        if self.tool == "buildah":
            return [
                "buildah",
                *self._storage_flags(),
                "bud",
                "--isolation",
                "chroot",
                *platform,
                "-t",
                tag,
                "-f",
                str(entry.context / "Dockerfile"),
                str(entry.context),
            ]
        return [self.tool, "build", *platform, "-t", tag, "-f", str(entry.context / "Dockerfile"), str(entry.context)]

    def push_argv(self, tag: str, digest_file: Path) -> list[str]:
        if self.tool == "buildah":
            return [
                "buildah",
                *self._storage_flags(),
                "push",
                "--digestfile",
                str(digest_file),
                tag,
                f"docker://{tag}",
            ]
        if self.tool == "podman":
            return ["podman", "push", "--digestfile", str(digest_file), tag]
        return ["docker", "push", tag]

    def inspect_argv(self, tag: str) -> list[str]:
        if self.tool == "buildah":
            return ["buildah", *self._storage_flags(), "inspect", "--type", "image", tag]
        return [self.tool, "image", "inspect", tag]

    # -- the pipeline --------------------------------------------------------------------------

    def _check(self, argv: list[str], what: str) -> subprocess.CompletedProcess:
        try:
            completed = self._run(argv)
        except OSError as exc:
            raise BuildError(f"{what} failed to start ({argv[0]}): {exc}") from exc
        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-15:])
            raise BuildError(f"{what} failed (exit {completed.returncode}): {' '.join(argv)}\n{tail}")
        return completed

    def build(self, entry: BuildPlanEntry) -> BuildResult:
        """Build, push when configured, inspect, and return the result to record. Raises :class:`BuildError`."""
        if not (entry.context / "Dockerfile").is_file():
            raise BuildError(f"{entry.context} has no Dockerfile")
        if self.tool == "buildah":
            self.storage_root.mkdir(parents=True, exist_ok=True)
        tag = self.tag_for(entry)
        tasks = ", ".join(entry.tasks[:5])
        self._check(self.build_argv(entry, tag), f"Building {tag} for task(s) {tasks}")
        record = self.inspect(tag)
        built_platform = f"{record.get('os')}/{record.get('architecture')}"
        if platform_key(built_platform) != platform_key(self.platform):
            raise BuildError(
                f"Building {tag} for task(s) {tasks}: the image came out as {built_platform}, not {self.platform}. "
                "Build on a host of that platform, or one with qemu/binfmt emulation; builds.platform "
                "overrides the target."
            )
        image = tag
        if self.push:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-push-") as tmp:
                digest_file = Path(tmp) / "digest"
                completed = self._check(self.push_argv(tag, digest_file), f"Pushing {tag}")
                digest = _pushed_digest(self.tool, digest_file, completed, self._run, tag)
            image = f"{tag}@{digest}"
        record = {**record, "image": image}
        return BuildResult(
            digest=entry.digest,
            tasks=entry.tasks,
            tag=tag,
            image=image,
            tool=self.tool,
            pushed=self.push,
            record=record,
        )

    def inspect(self, tag: str) -> dict[str, Any]:
        """The image's OCI configuration in the ``compose-images.json`` record shape (``image`` unset)."""
        completed = self._check(self.inspect_argv(tag), f"Inspecting {tag}")
        try:
            data = json.loads(completed.stdout)
        except ValueError as exc:
            raise BuildError(f"Inspecting {tag}: {self.tool} returned no JSON ({exc})") from exc
        return normalize_inspect(self.tool, data)


def _pushed_digest(tool: str, digest_file: Path, completed: subprocess.CompletedProcess, run: Runner, tag: str) -> str:
    """The manifest digest the registry assigned to the pushed image."""
    if tool in ("buildah", "podman"):
        if digest_file.is_file():
            digest = digest_file.read_text().strip()
            if digest.startswith("sha256:"):
                return digest
        raise BuildError(f"Pushing {tag}: {tool} wrote no digest file")
    # docker: the push output ends with `latest: digest: sha256:... size: ...`; RepoDigests holds it afterwards.
    for line in (completed.stdout or "").splitlines():
        if "digest: sha256:" in line:
            return "sha256:" + line.split("digest: sha256:", 1)[1].split()[0]
    inspected = run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", tag])
    if inspected.returncode == 0:
        for repo_digest in json.loads(inspected.stdout or "[]"):
            if "@sha256:" in repo_digest:
                return repo_digest.split("@", 1)[1]
    raise BuildError(f"Pushing {tag}: docker reported no digest")


def normalize_inspect(tool: str, data: Any) -> dict[str, Any]:
    """Turn a tool's inspect JSON into the ``compose-images.json`` record shape (without ``image``).

    buildah reports ``{"OCIv1": {"architecture", "os", "config"}, "FromImageID"}``; podman and docker
    report a list with ``Architecture``, ``Os``, ``Config`` and ``Id``.
    """
    if tool == "buildah":
        if not isinstance(data, dict) or "OCIv1" not in data:
            raise BuildError("buildah inspect returned no OCIv1 block")
        oci = data["OCIv1"]
        config = oci.get("config") or {}
        config_digest = data.get("FromImageID") or ""
        architecture, os_name = oci.get("architecture"), oci.get("os")
    else:
        entry = data[0] if isinstance(data, list) and data else data
        if not isinstance(entry, dict) or "Config" not in entry:
            raise BuildError(f"{tool} image inspect returned no Config block")
        config = entry.get("Config") or {}
        config_digest = str(entry.get("Id") or "")
        architecture, os_name = entry.get("Architecture"), entry.get("Os")
    if config_digest and not config_digest.startswith("sha256:"):
        config_digest = f"sha256:{config_digest}"
    if (os_name, architecture) != ("linux", "amd64"):
        LOGGER.warning(
            "Built image is %s/%s, not linux/amd64; it runs only on a sandbox host of that platform",
            os_name,
            architecture,
        )
    return {
        "architecture": architecture,
        "config": {key: config[key] for key in RECORDED_CONFIG_KEYS if config.get(key)},
        "config_digest": config_digest,
        "os": os_name,
    }


# -- recording -----------------------------------------------------------------------------------


def platform_key(platform: str | None) -> str:
    """``os/arch`` of a platform string, with a variant such as ``linux/arm64/v8`` dropped."""
    parts = (platform or "").split("/")
    return "/".join(parts[:2])


def recorded_image_mismatch(
    entry: dict[str, Any], image_configs: dict[str, Any], *, repository: str | None, platform: str
) -> str | None:
    """Why a recorded built image cannot serve this builder, or ``None`` when it can.

    A local (unpushed) image serves only local builds; a pushed one serves both. The recorded
    platform has to be the one asked for.
    """
    if not entry.get("pushed") and repository is not None:
        return "it was built locally and this run pushes to a repository"
    record = image_configs.get(entry.get("image") or "")
    if record is None:
        return "its configuration is missing from compose-images.json"
    recorded = f"{record.get('os')}/{record.get('architecture')}"
    if platform_key(recorded) != platform_key(platform):
        return f"it is {recorded}, not {platform}"
    return None


def record_built_image(folder: Path, result: BuildResult, *, overwrite: bool = False) -> Path:
    """Record one built image next to the task folders: its context digest in ``built-images.json``
    and its configuration in ``compose-images.json``. Existing entries are kept unless ``overwrite``."""
    folder = Path(folder)
    built_path = folder / BUILT_IMAGES_FILE
    built = read_built_images(folder)
    if result.digest in built and not overwrite:
        # Pinned: a context recorded once keeps its image, like compose-images.json keeps its entries.
        return built_path
    built[result.digest] = {
        "image": result.image,
        "tasks": sorted(result.tasks),
        "tool": result.tool,
        "pushed": result.pushed,
    }
    configs_path = folder / IMAGE_CONFIGS_FILE
    configs: dict[str, Any] = json.loads(configs_path.read_text()) if configs_path.is_file() else {}
    if overwrite:
        configs[result.image] = result.record
    else:
        configs.setdefault(result.image, result.record)
    configs_path.write_text(json.dumps(dict(sorted(configs.items())), indent=2, sort_keys=True) + "\n")
    built_path.write_text(json.dumps(dict(sorted(built.items())), indent=2, sort_keys=True) + "\n")
    return built_path


def build_images(
    folder: Path,
    *,
    builder: ImageBuilder,
    select: Callable[[str], bool] | None = None,
    log: Callable[[str], None] = print,
) -> list[BuildResult]:
    """Build and record every context under ``folder`` that needs it and has no usable record.

    A recorded image is reused when it fits this builder: pushed, or local when nothing is pushed,
    and of the requested platform. Otherwise it is rebuilt and its record replaced, and the reason
    is logged. A context is keyed by its content hash, so a changed build context is a new build;
    deleting a context's entry from ``built-images.json`` forces one for an unchanged context.
    Returns the images built by this call. Raises :class:`BuildError` on the first failure, after
    recording whatever was built before it.
    """
    folder = Path(folder).resolve()
    parent = folder.parent if is_task_folder(folder) else folder
    plan = plan_builds(folder, select=select)
    if not plan:
        return []
    recorded = read_built_images(parent)
    image_configs = read_image_configs(parent)
    results: list[BuildResult] = []
    for entry in plan:
        names = ", ".join(entry.tasks[:3])
        if entry.digest in recorded:
            why = recorded_image_mismatch(
                recorded[entry.digest], image_configs, repository=builder.repository, platform=builder.platform
            )
            if why is None:
                log(f"Built image recorded for {names}: {recorded[entry.digest]['image']}")
                continue
            log(f"Rebuilding {recorded[entry.digest]['image']} for {names}: {why}")
        result = builder.build(entry)
        # The build just made is authoritative for its image: a configuration left behind under the same
        # name (by a deleted built-images.json entry, or a record that no longer fits) is replaced.
        record_built_image(parent, result, overwrite=True)
        where = "pushed" if result.pushed else "local"
        log(f"Built {result.image} ({where}, {result.tool}) for task(s) {', '.join(entry.tasks[:5])}")
        results.append(result)
    return results
