# RFSoC\_SDR

*A Software‑Defined Radio (SDR) framework for millimetre‑wave channel sounding on the Xilinx RFSoC 4×2 platform.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [RFSoC 4×2 Setup](#rfsoc-4×2-setup)
4. [Host Computer Setup](#host-computer-setup)
5. [Pi‑Radio FR3 Transceiver Setup](#pi‑radio-fr3-transceiver-setup)
6. [Running Measurements](#running-measurements)
7. [Citation](#citation)

---

## Project Overview

This repository provides all hardware descriptions, Python utilities and example notebooks needed to perform wide‑band millimetre‑wave measurements with a Xilinx RFSoC 4×2 board and a Pi‑Radio FR3 transceiver. The system has been verified in the 57–64 GHz band with Vivaldi antennas and can be adapted to other frequency ranges with minor clock‑plan and filter changes.

> **Why another SDR?**  Typical COTS SDRs top out below 6 GHz or suffer from limited instantaneous bandwidth at mmWave. The RFSoC 4×2 couples 5 Gsps DAC/ADC channels directly to the FPGA fabric, enabling real‑time TDD beam‑sounding and raw I/Q capture without an external front‑end.

---

## Prerequisites

Hardware

- Xilinx **RFSoC 4×2** board
- **Pi‑Radio FR3** transceiver
- Two **Vivaldi antennas** with printable spacers (to set boresight spacing)
- High‑quality **SMA ↔ SMA** cables (if antennas are not connected directly)
- **DC blockers** and **DC‑1300 MHz filters** (see wiring tables below)
- **20 dB fixed attenuators** for the receive path
- 16 GB (or larger) **micro‑SD card**
- USB‑C cable (console + UART)
- Bench supply (12 V / 6 A recommended)

Software

- **Git** ≥ 2.34
- **Python** 3.10 with `pip` and `venv`
- **Vivado** 2023.2 *only if you plan to rebuild the bitstream*
- Internet access to download board images and Python wheels

---

## RFSoC 4×2 Setup

### 1 · Flash the SD‑card image

1. Download the latest RFSoC 4×2 PYNQ image (tested with **v3.0.1**) from the [official releases page](https://www.pynq.io/boards.html).
2. Use **Rufus** (Win), **balenaEtcher** (macOS/Linux) or similar to write the `.img` file to the micro‑SD card.
3. Safely eject the card and insert it into the RFSoC.

### 2 · First boot

1. Connect USB‑C to your PC for serial console and power the board.
2. Wait \~90 s until the OLED/LCD shows an IP address (default DHCP on `usb0` is **192.168.3.1**).
3. Browse to [**http://192.168.3.1:9090/lab**](http://192.168.3.1:9090/lab) to confirm the Jupyter server is alive.

### 3 · Install project files on‑board

```bash
# SSH into the board (default password is xilinx)
ssh xilinx@192.168.3.1

# Make a workspace for notebooks
mkdir -p ~/jupyter_notebooks/RFSoC_SDR && cd $_

# Clone only the python helpers (saves time)
svn export https://github.com/ali-rasteh/RFSoC_SDR/trunk/python ./python

# Copy clock configuration files required by xrfclk
sudo cp -r python/rfsoc/rfsoc4x2_clock_configs \
        $(python - <<'PY'
import site, pathlib, sys
print(pathlib.Path(site.getsitepackages()[0])/'xrfclk')
PY)/

# Copy the pre‑built bitstream (optional—skip if you will build your own)
svn export https://github.com/ali-rasteh/RFSoC_SDR/trunk/vivado/sounder_fr3_if_ddr4_mimo_4x2/builds/ .
```

> **Note**  Set `import_pynq = True` and `import_sivers = True` in `backend.py` if you intend to use the Sivers front‑end APIs.

### 4 · Install RFSoC‑MTS overlay

Follow the official instructions in the [Xilinx RFSoC‑MTS repo](https://github.com/Xilinx/RFSoC-MTS#installation). The MTS driver provides sub‑sample TX/RX alignment that our channel‑sounder notebooks rely on.

---

## Host Computer Setup

> All host commands presume **bash/zsh** on Linux/macOS or **PowerShell** on Windows. Replace paths as needed.

```bash
# 1 · Clone repo
$ git clone https://github.com/ali-rasteh/RFSoC_SDR.git && cd RFSoC_SDR

# 2 · Create isolated Python env
$ python -m venv env
$ source env/bin/activate   # PowerShell: .\env\Scripts\Activate.ps1

# 3 · Install dependencies
(env) $ pip install -r requirements.txt
```

> **Updating deps**  After adding packages, run `pip freeze > requirements.txt`. Delete any `pywin32‑*` lines before committing—those wheels are Windows‑only.

Exit the venv with `deactivate`.

---

## Pi‑Radio FR3 Transceiver Setup

### 1 · Hardware connections

| Path                       | Component          | Inline Parts                                        |
| -------------------------- | ------------------ | --------------------------------------------------- |
| **TX** (RFSoC→Transceiver) | DAC\_A → IF Port 1 | DC blocker → DC‑1300 MHz LPF                        |
|                            | DAC\_B → IF Port 2 | DC blocker → DC‑1300 MHz LPF                        |
| **RX** (Transceiver→RFSoC) | IF Port 1 → ADC\_B | DC blocker → DC‑1300 MHz LPF → **20 dB attenuator** |
|                            | IF Port 2 → ADC\_D | DC blocker → DC‑1300 MHz LPF → **20 dB attenuator** |

> **Direct‑mount option**  If the antennas bolt straight to the RF ports, the DC blockers & filters may be omitted *provided* the DUT’s base‑band path is DC‑isolated.

### 2 · Bring‑up scripts

```bash
# Default login
ssh ubuntu@192.168.137.51   # pwd: temppwd

# One‑shot initialisation (sets LO, bias, cal tables)
./do_everything.sh
```

You can also configure the board via its Bokeh GUI at [http://192.168.137.51:5006](http://192.168.137.51:5006).

---

## Running Measurements

1. **Launch the server** on the RFSoC:

   ```bash
   # In a Jupyter notebook cell or SSH session
   python ~/jupyter_notebooks/RFSoC_SDR/python/rfsoc_test.py
   ```

   The script prints `Waiting for a connection…` once the RX/TX DMA engines are ready.

2. **Configure and start a sweep** from your host PC:

   ```bash
   (env) $ python python/rfsoc_test.py --cfg configs/60ghz_sweep.yaml
   ```

   - The first run prompts you to perform a phase‑cal procedure; follow the on‑screen instructions.
   - Live constellations and CIR plots appear via `matplotlib`.
   - Append items to the `save_list` parameter in your YAML to log raw HDF5 dumps for later analysis.

---

## Citation

If this work assists your research, please cite:

```bibtex
@misc{Rasteh_RFSoC_SDR_2024,
  author       = {Ali Rasteh},
  title        = {RFSoC\_SDR — Millimetre‑Wave Channel Sounder},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ali-rasteh/RFSoC_SDR}},
  doi          = {10.5281/zenodo.14846067},
  note         = {Accessed: 16 Jun 2024}
}
```

---

© 2024 Ali Rasteh • Released under the MIT License

