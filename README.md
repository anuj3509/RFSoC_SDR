# RFSoC_SDR

A Software Defined Radio implementation using Xilinx RFSoC4x2 for millimeter-wave measurements.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [RFSoC4x2 Setup](#rfsoc4x2-setup)
3. [Host Computer Setup](#host-computer-setup)
4. [Pi-Radio FR3 Transceiver Setup](#pi-radio-fr3-transceiver-setup)
5. [Running Measurements](#running-measurements)
6. [Citation](#citation)

## Prerequisites

- RFSoC4x2 board
- Pi-Radio FR3 Transceiver
- Vivaldi antennas with appropriate spacers
- SD card (for RFSoC4x2)
- USB cable
- Internet connection
- DC blockers and filters:
  - DC-1300 MHz filters
  - 20dB attenuators

## RFSoC4x2 Setup

### Initial Setup
1. Download the RFSoC4x2 image from [PYNQ.io](https://www.pynq.io/boards.html)
   - Tested with v3.0.1
   - Newer versions should work unless not backward compatible

2. Program the SD card:
   - Use Rufus (Windows) or similar software for your OS
   - Write the RFSoC4x2 image to the SD card

3. Boot the RFSoC4x2:
   - Insert the SD card
   - Power on the board
   - Wait for IP address display on LCD
   - Connect via USB cable

4. Access the web interface:
   - Open browser and navigate to: http://192.168.3.1:9090/lab/

### Software Setup
1. Create project directory:
   ```bash
   mkdir /home/xilinx/jupyter_notebooks/YOURFOLDER/
   ```

2. Copy required files:
   - Copy Python scripts from [python folder](https://github.com/ali-rasteh/RFSoC_SDR/tree/main/python)
   - Copy clock configuration files from [rfsoc4x2_clock_configs](https://github.com/ali-rasteh/RFSoC_SDR/tree/main/rfsoc/rfsoc4x2_clock_configs) to:
     ```
     /usr/local/share/pynq-venv/lib/python3.10/site-packages/xrfclk/
     ```
   - Copy FPGA image files (`.bit` and `.hwh` only) from [builds folder](https://github.com/ali-rasteh/RFSoC_SDR/tree/main/vivado/sounder_fr3_if_ddr4_mimo_4x2/builds)

3. Configure `backend.py`:
   - Set `import_pynq = True`
   - Set `import_sivers = True` if using Sivers antenna

4. Install RFSoC-MTS:
   - Follow instructions from [RFSoC-MTS repository](https://github.com/Xilinx/RFSoC-MTS/tree/main)

## Host Computer Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ali-rasteh/RFSoC_SDR.git
   cd RFSoC_SDR
   ```

2. Create and activate virtual environment:
   ```bash
   # Create environment
   python -m venv env

   # Activate environment
   # Windows:
   .\env\Scripts\Activate.ps1
   # If you get a security error, run:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   
   # Mac/Linux:
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. To exit virtual environment:
   ```bash
   deactivate
   ```

### Managing Requirements
To update requirements.txt:
```bash
python -m pip freeze > requirements.txt
```
Note: On Windows, remove any `pywin32` entries from requirements.txt as they are Windows-specific.

## Pi-Radio FR3 Transceiver Setup

1. Hardware Setup:
   - Assemble Vivaldi antennas with appropriate spacers to tune the antenna spacing needed for the target frequency
   - Mount antennas on fixed structure
   - Connect RF ports of the Pi-Radio FR3 transceiver to Vivaldi antennas
   - Connect IF ports of the Pi-Radio FR3 transceiver to RFSoC4x2:
     - RFSoC4x2 Output: Port 1 (DAC_A), Port 2 (DAC_B)
     - RFSoC4x2 Input: Port 1 (ADC_B), Port 2 (ADC_D)
   - Add required components(only if you are using cables):     # not needed when you are connecting antenna directly to RF port
     - Transmitter(RFSoC4x2 ports): DC blocker and a DC-1300 MHz filter
     - Receiver: DC blocker, a DC-1300 MHz filter and a 20dB attenuator on each port

2. Power and Configure:
   - Power on the Pi-Radio FR3 Transceiver
   - Connect to board:
     ```bash
     ssh ubuntu@192.168.137.51
     # Password: temppwd
     ```
   - Run configuration script:
     ```bash
     ./do_everything.sh
     ```
   - Alternative: Use web GUI at http://192.168.137.51:5006

## Running Measurements

1. Start RFSoC4x2:
   - Connect to laptop/PC via USB cable
   - Access web interface at http://192.168.3.1:9090/lab/
   - Run `rfsoc_test.py` using a jupyter notebook or by using the below command:
     ```bash
     python rfsoc_test.py
     ```
   - Wait for "Waiting for a connection" message

2. Run measurements:
   - Configure `rfsoc_test.py` on your host computer/PC
   - Run the script in an environment with plotting capabilities, FYI we are using matplotlib (recommended: VSCode)
   - If doing for the first time, follow phase calibration instructions on the receiver side when prompted
   - View results in animation plots
   - Save data by adding elements to `save_list` parameter

## Citation

If you use this repository or code in your research, please cite it as follows:

### BibTeX
```bibtex
@misc{Rasteh_RFSoC_SDR,
  author       = {Rasteh, Ali},
  title        = {Software Defined Radio using Xilinx RFSoC},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ali-rasteh/RFSoC_SDR}},
  doi          = {[![DOI](https://zenodo.org/badge/821517620.svg)](https://doi.org/10.5281/zenodo.14846067)},
  note         = {Accessed: 2024-06-16}
}
```
