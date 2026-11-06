"""Tự động giải nén dữ liệu từ dataverse_files và tải các file dữ liệu gốc còn thiếu."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

DATAVERSE_MAPPING = {
    "HCMC_Floods_BDD.gpkg": "HCMC_floods/Layer_files/HCMC_Floods_BDD.gpkg",
    "HCMC_new_limits.gpkg": "HCMC_floods/Layer_files/HCMC_new_limits.gpkg",
    "HCMC_water.gpkg": "HCMC_floods/Layer_files/HCMC_water.gpkg",
    "Metadata-form-HCMC-floods.pdf": "HCMC_floods/Metadata-form-HCMC-floods.pdf",
    "ReadMe_10_23708_8Y16HU.txt": "ReadMe_10_23708_8Y16HU.txt",
}

DIRECT_DOWNLOADS = {
    "vietnam-latest.osm.pbf": "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf",
    "ESA_WorldCover_10m_2021_v200_N09E105_Map.tif": (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        "v200/2021/map/ESA_WorldCover_10m_2021_v200_N09E105_Map.tif"
    ),
}

FABDEM_ZIP_URL = (
    "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
    "N10E100-N20E110_FABDEM_V1-2.zip"
)
FABDEM_TILES = {
    "N10E106_FABDEM_V1-2.tif": "fabdem_tiles/N10E106_FABDEM_V1-2.tif",
    "N10E107_FABDEM_V1-2.tif": "fabdem_tiles/N10E107_FABDEM_V1-2.tif",
    "N11E106_FABDEM_V1-2.tif": "fabdem_tiles/N11E106_FABDEM_V1-2.tif",
    "N11E107_FABDEM_V1-2.tif": "fabdem_tiles/N11E107_FABDEM_V1-2.tif",
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Đang tải: {url} -> {destination} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, tmp_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file, length=1024 * 1024)
    tmp_path.replace(destination)
    print(f"Hoàn tất tải: {destination.name} ({destination.stat().st_size:,} bytes)")


def prepare_raw_directory(raw_root: Path, manifest_path: Path | None = None) -> None:
    raw_root = raw_root.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)

    # 1. Trích xuất 5 file từ dataverse_files (nếu có)
    dv_dir = raw_root / "dataverse_files"
    if dv_dir.is_dir():
        for dst_name, rel_src in DATAVERSE_MAPPING.items():
            src_file = dv_dir / rel_src
            dst_file = raw_root / dst_name
            if src_file.is_file() and not dst_file.is_file():
                shutil.copy2(src_file, dst_file)
                print(f"Đã copy từ dataverse_files: {dst_name}")

    # 2. Tải trực tiếp OSM PBF và ESA WorldCover nếu chưa có
    for rel_path, url in DIRECT_DOWNLOADS.items():
        target = raw_root / rel_path
        if not target.is_file():
            _download_file(url, target)

    # 3. Tải và giải nén 4 tile FABDEM v1.2 nếu chưa có đủ
    missing_tiles = {
        tile_name: raw_root / rel_path
        for tile_name, rel_path in FABDEM_TILES.items()
        if not (raw_root / rel_path).is_file()
    }
    if missing_tiles:
        (raw_root / "fabdem_tiles").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=raw_root) as tmp_dir:
            zip_path = Path(tmp_dir) / "FABDEM.zip"
            _download_file(FABDEM_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    member_name = Path(member.filename).name
                    if member_name in missing_tiles:
                        dst = missing_tiles[member_name]
                        with zf.open(member) as src, dst.open("wb") as out:
                            shutil.copyfileobj(src, out, length=1024 * 1024)
                        print(f"Đã giải nén FABDEM tile: {dst.relative_to(raw_root)}")

    # 4. Cập nhật lại bytes & sha256 trong manifest.yaml nếu file tải mới có kích thước khác
    if manifest_path is not None and manifest_path.is_file():
        doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        changed = False
        for dataset in doc.get("datasets", []):
            for asset in dataset.get("assets", []):
                file_path = raw_root / asset["path"]
                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"Vẫn thiếu file dữ liệu gốc bắt buộc: {file_path}"
                    )
                size = file_path.stat().st_size
                with file_path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if asset.get("bytes") != size or asset.get("sha256") != digest:
                    print(
                        f"Cập nhật checksum mới cho {asset['path']} "
                        f"({size:,} bytes, sha256={digest[:12]}...)"
                    )
                    asset["bytes"] = size
                    asset["sha256"] = digest
                    changed = True
        if changed:
            manifest_path.write_text(
                yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            print(f"Đã đồng bộ checksum mới vào {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Thư mục chứa dữ liệu gốc (mặc định: data/raw)",
    )
    parser.add_argument(
        "--manifest",
        default="data/raw/manifest.yaml",
        help="Đường dẫn tới file manifest.yaml cần đồng bộ checksum",
    )
    args = parser.parse_args()
    prepare_raw_directory(Path(args.raw_root), Path(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

