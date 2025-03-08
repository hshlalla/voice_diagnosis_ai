# English

## Folder Structure

**voice_diagnosis_api (Root Directory)**

- `/data` - Stores data files.
- `/docs` - Code history requiring version control.
- `/src` - Contains all source code, including Python scripts and web-related files.
  - Docker setup available for environment installation.
  - Includes Python files: `web.py`, `diagnosis.py`, `main.py`, `configure.py`, `utils.py`.
  - `models` folder - Place the downloaded model files here.
  - `static` & `templates` folders - Related to the web interface.
- `/train` - Contains training scripts for model creation.

## Model Information

- The model required for inference is **not uploaded to Git** due to its size.
- Please place the **model folder inside the `src/models` directory**.

## How to Use the API

1. Start the Flask web server:
   ```bash
   cd src 
   python main.py
2. Access the API endpoints:
- Upload and analyze audio files via the /diagnosis endpoint.
- Check the API version via the /version endpoint.

## How to Access via Web Browser

1. Start the server as described above.
2. Open a web browser and go to:
- http://server-ip:9090

3. Use the provided interface to upload voice files and receive analysis results.

model downlink
https://webhard.aible.kro.kr/share/HWzw9BCG
