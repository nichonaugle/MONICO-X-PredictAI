# MONICO-X-PredictAI

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [API](#api)
4. [Networking](#testing)
5. [Contributing](#contributing)

## Installation

1. **Cloning the Repository**:
   ```bash
   git clone https://github.com/nichonaugle/MONICO-X-PredictAI
   cd MONICO-X-PREDICTAI
   ```

2. **Setting Up the Virtual Environment**:
   - Ensure your Python version is 3.11.
   - Install virtualenv if it's not already installed: `pip install virtualenv`.
   - Create and activate the virtual environment:
     - For Windows: `python -m venv venv` and `venv/Scripts/activate`.
     - For macOS/Linux: `python -m venv venv` and `source venv/bin/activate`.

3. **Installing Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Before running any commands, ensure you are inside the virtual environment.

## API

The API provides the following endpoints:

- `GET /server/active-tasks`: Retrieves the active tasks on the server.
- `GET /server/set-streaming`: Sets the data live streaming state.
- `GET /server/info`: Retrieves the device information.
- `GET /data/set-sending-interval-period`: Sets the data sending interval period.
- `GET /ai/set-prediction-state`: Sets the AI prediction state.


## Networking
The config.json file includes several networking parameters used to configure mDNS on the local network:

- `"MDNS_DEVICE_NAME": "predictai-XXXXX"`
- `"MDNS_SERVICE_TYPE": "_predictai-ws._tcp.local."`
- `"SERVER_PORT": "8000"`
- `"MDNS_SERVICE_NAME": "Monico PredictAI Hub"`

If you leave the MDNS_DEVICE_NAME as predictai-XXXXX, the device will automatically generate and resolve its hostname via mDNS.
To access the FastAPI interface and test API endpoints, navigate to:
http://<MDNS_DEVICE_NAME>:<SERVER_PORT>/docs

## Contributing

To contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.