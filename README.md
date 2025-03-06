# Fine-Tuning-F5-TTS-Hindi
Fine tuning F5 tts on indic voices-r data for hindi language </br>
## Installation

### Create a separate environment if needed

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5-tts python=3.10
conda activate f5-tts
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
> ```

</details>

Trained on 4060 Ti 8GB VRAM</br>
25989 samples audio length vary from 4s-10second </br>


![Screenshot from 2025-03-06 11-47-54](https://github.com/user-attachments/assets/3500c6a2-4a0c-4bad-b1a1-aaa2c6b5c02f)
![Screenshot from 2025-03-06 11-48-04](https://github.com/user-attachments/assets/41eec914-7238-4920-bdb5-a36962d093c2)
![Screenshot from 2025-03-06 11-48-20](https://github.com/user-attachments/assets/a495c954-3b6e-4b5f-8d64-954a06bfbf3d)
