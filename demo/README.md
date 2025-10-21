---
title: Interactive demonstration of minimum Bayes risk decoding for automatic speech recognition 
emoji: 🐨
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
pinned: false
license: MIT License
---

This code runs Gradio app for comparing beam search and MBR decoding on the sound input via a file or a microphone.
The default task is to transcribe English speech in the audio.
It would be useful to quickly evaluate the gain of MBR decoding interactively. 

The app runs with the following command and accesible by a browser.
It runs on GPU if available and otherwise CPU. 

```
python3 app.py
```
