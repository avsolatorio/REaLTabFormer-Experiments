from pathlib import Path
import shutil
from config_lib import load_config


EXP_DIR = Path(__file__).parent.parent / "exp"
print(EXP_DIR)

src_config = EXP_DIR / "base_rtf_config.toml"

conf = src_config.read_text()

for data_path in EXP_DIR.glob("*"):
    if data_path.is_file():
        continue

    rtf_dir = (data_path / "realtabformer")

    rtf_dir.mkdir(parents=True, exist_ok=True)

    # if (rtf_dir / "config.toml").exists():
    #     continue
    # else:

    cconf = str(conf)
    cconf = cconf.replace("/DATA_ID/", f"/{data_path.name}/")
    (rtf_dir / "config.toml").write_text(cconf)

    # shutil.copy2(src_config, rtf_dir / "config.toml")
    assert (rtf_dir / "config.toml").exists()
