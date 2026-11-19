"""Create or update the Docker Space using just the files that run the demo."""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]


def files_to_publish():
    files = {
        "README.md": ROOT / "space" / "README.md",
        "Dockerfile": ROOT / "Dockerfile",
        ".dockerignore": ROOT / ".dockerignore",
        "pyproject.toml": ROOT / "pyproject.toml",
    }
    package = ROOT / "src" / "speech_translate"
    for pattern in ("*.py", "web/*.js", "web/*.html", "web/*.css", "web/example.wav",
                    "web/example-hawk.wav", "web/example-mountain.wav", "web/example-arrival.wav"):
        paths = sorted(package.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"Missing demo source or asset: {pattern}")
        for path in paths:
            files[path.relative_to(ROOT).as_posix()] = path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Your HF namespace and Space name")
    parser.add_argument("--dry-run", action="store_true", help="Print the upload manifest")
    parser.add_argument("--update", action="store_true", help="Update an existing Docker Space")
    args = parser.parse_args()
    files = files_to_publish()
    if args.dry_run:
        print("\n".join(files))
        return

    api = HfApi()
    if not args.update:
        api.create_repo(args.repo_id, repo_type="space", space_sdk="docker",
                        space_hardware="cpu-basic", private=False)
    space = api.space_info(args.repo_id)
    if space.sdk != "docker":
        raise ValueError(f"{args.repo_id} uses {space.sdk}, expected docker")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                  for name, path in files.items()]
    commit = api.create_commit(repo_id=args.repo_id, repo_type="space", operations=operations,
                               parent_commit=space.sha, commit_message="Deploy LiveTranslate speech demo")
    print(commit.commit_url)
    print(f"https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
