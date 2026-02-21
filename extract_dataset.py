import argparse
import zipfile
from pathlib import Path

ACTIONS = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]


def extract_action(zip_path: Path, class_dir: Path) -> None:
    """Extract all .avi files from a zip into class_dir (flat, no subfolders)."""

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    class_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    existing = list(class_dir.glob("*.avi"))
    if existing:
        print(f"  [SKIP] {class_dir.name}/ already has {len(existing)} AVI files.")
        return

    print(f"  [EXTRACT] {zip_path.name}  →  {class_dir}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        avi_members = [m for m in zf.namelist() if m.lower().endswith(".avi")]
        for member in avi_members:
            filename = Path(member).name          # strip any subfolder in zip
            dest = class_dir / filename
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    count = len(list(class_dir.glob("*.avi")))
    print(f"           → {count} AVI files extracted.")


def main(zip_dir: Path, out_dir: Path) -> None:
    print(f"Zip source : {zip_dir.resolve()}")
    print(f"Output root: {out_dir.resolve()}\n")

    for action in ACTIONS:
        zip_path  = zip_dir / f"{action}.zip"
        class_dir = out_dir / action
        extract_action(zip_path, class_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Extraction complete. File counts per class:")
    for action in ACTIONS:
        n = len(list((out_dir / action).glob("*.avi")))
        print(f"  {action:15s}: {n:>4d} videos")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_dir", default="data",
                        help="Folder containing the 6 zip files (default: data/)")
    parser.add_argument("--out_dir", default="data/kth_actions",
                        help="Output root folder for extracted videos (default: data/kth_actions/)")
    args = parser.parse_args()

    main(Path(args.zip_dir), Path(args.out_dir))