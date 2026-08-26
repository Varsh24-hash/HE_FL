import json
import zipfile
import tempfile
import os
INPUT_ZIP = "entity_keys/client_key.zip"
OUTPUT_ZIP = "entity_keys/public_key.zip"
def remove_secret_parts(data):
    if "keyset" in data:
        ks = data["keyset"]
        ks.pop("lweSecretKeys", None)
        ks.pop("lweBootstrapKeys", None)
        ks.pop("lweKeyswitchKeys", None)
        ks.pop("packingKeyswitchKeys", None)
    return data
def create_public_key_zip():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract ZIP
        with zipfile.ZipFile(INPUT_ZIP, "r") as z:
            z.extractall(tmpdir)
        # DEBUG (optional but useful)
        print("Extracted files:")
        for root, _, files in os.walk(tmpdir):
            for f in files:
                print(" -", f)
        # Find client.specs (NOW DIRECTLY INSIDE ZIP)
        specs_path = None
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f == "client.specs.json":
                    specs_path = os.path.join(root, f)
                    break
        if not specs_path:
            raise FileNotFoundError("client.specs not found inside zip")
        print("Found specs at:", specs_path)
        # Load specs
        with open(specs_path, "r") as f:
            data = json.load(f)
        # Remove secrets
        public_data = remove_secret_parts(data)
        # Write cleaned file
        public_json = os.path.join(tmpdir, "public.specs.json")
        with open(public_json, "w") as f:
            json.dump(public_data, f, indent=2)
            json.dump(public_data, f, indent=2)
        # Re-zip
        with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(public_json, arcname="public.specs.json")
    print("Created:", OUTPUT_ZIP)
if __name__ == "__main__":
    create_public_key_zip()

