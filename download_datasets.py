import os
import sys
import time
import urllib.request
import zipfile

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

def download_file_with_retry(url, target_path, expected_size=None, max_retries=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    
    retries = 0
    while retries < max_retries:
        existing_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        if expected_size and existing_size >= expected_size:
            print(f"Archivo completo ya descargado ({existing_size / (1024*1024):.1f} MB).")
            return True
            
        req = urllib.request.Request(url, headers=headers)
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")
            print(f"\nReanudando descarga desde el byte {existing_size} ({existing_size / (1024*1024):.1f} MB)...")
            mode = "ab"
        else:
            mode = "wb"
            
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status_code = resp.getcode()
                content_range = resp.headers.get("Content-Range")
                total_len = resp.headers.get("Content-Length")
                
                # Si el servidor no soportó Range y devolvió 200 en vez de 206, reiniciamos desde 0
                if existing_size > 0 and status_code == 200:
                    print("El servidor no soporta Range; reiniciando descarga completa...")
                    existing_size = 0
                    mode = "wb"
                    
                if content_range:
                    total_size = int(content_range.split("/")[-1])
                elif total_len:
                    total_size = existing_size + int(total_len)
                else:
                    total_size = expected_size or -1
                    
                downloaded = existing_size
                block_size = 1024 * 1024 # 1MB
                
                with open(target_path, mode) as f:
                    while True:
                        chunk = resp.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100.0
                            sys.stdout.write(f"\rDescargando: {pct:5.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                        else:
                            sys.stdout.write(f"\rDescargando: {downloaded / (1024*1024):.1f} MB")
                        sys.stdout.flush()
                        
                if total_size > 0 and downloaded < total_size:
                    print(f"\nConexión interrumpida en {downloaded / (1024*1024):.1f} MB de {total_size / (1024*1024):.1f} MB. Reintentando...")
                    retries += 1
                    time.sleep(2)
                    continue
                else:
                    print(f"\nDescarga completa exitosa ({downloaded / (1024*1024):.1f} MB).")
                    return True
                    
        except urllib.error.HTTPError as e:
            if e.code == 416: # Range Not Satisfiable -> ya está completo
                print("\nEl rango solicitado ya excede el archivo (posiblemente completo).")
                return True
            print(f"\nHTTP Error {e.code}: {e.reason}. Reintentando...")
            retries += 1
            time.sleep(3)
        except Exception as e:
            print(f"\nError de red: {e}. Reintentando en 3s...")
            retries += 1
            time.sleep(3)
            
    return False

def download_glvd():
    glvd_dir = os.path.join(DATASETS_DIR, "GLVD")
    os.makedirs(glvd_dir, exist_ok=True)
    zip_path = os.path.join(DATASETS_DIR, "GLVD.zip")
    url = "https://zenodo.org/api/records/18937397/files/Grapevine%20Leaf%20Variety%20&%20Disease%20Dataset%20(GLVD).zip/content"
    total_expected = 1104204711 # 1053.05 MB segun Zenodo API
    
    print("=" * 60)
    print("Iniciando descarga robusta de GLVD...")
    print(f"URL: {url}")
    print(f"Destino: {zip_path}")
    print(f"Tamaño esperado: {total_expected / (1024*1024):.1f} MB")
    print("=" * 60)
    
    success = download_file_with_retry(url, zip_path, expected_size=total_expected)
    if not success:
        print("Error: No se pudo completar la descarga tras varios reintentos.")
        sys.exit(1)
        
    print(f"\nDescomprimiendo en: {glvd_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(glvd_dir)
    print("\n¡Dataset GLVD descargado y descomprimido con éxito!")

if __name__ == "__main__":
    os.makedirs(DATASETS_DIR, exist_ok=True)
    download_glvd()
