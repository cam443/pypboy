# pypboy

> _Notes from ZapWizard:_
>
> This is a work in progress of the code for my Functional Pip-Boy 3000 MK IV.
>
> I branched off from the Fallout 3 style Pip-Boy 3000 code.
> The graphics are positioned for a 720x720 display.

## Controls

#### Navigation

|     |             |
| --- | ----------- |
| F1  | Stats       |
| F2  | Inventory   |
| F3  | Data        |
| F4  | Map         |
| F5  | Radio       |
| F6  | Boot screen |

1,2,3,4, etc. are used to navigate the sub-header menu.

Up/Down arrow keys to navigate the sub-menus.

+/- to zoom the map

## Adding Radio Stations

Adding radio stations is now as easy as making a folder in sounds/radio with MP3, OGG or WAV files.
Make a file named `Station.py`, with `station_name = "Your name here"` to set the menu text. The folder name is used if this file is missing.
Add a number to the beginning of the folder name to set the menu position.

If you want the original in-game music you can use the B.A.E. program to extract the files from the game, then the Yakitori Audio Converter to convert them back to MP3 files.

## Developing

The easiest way to keep track of local development is by creating a Python virtual environment, create one using `python -m venv venv`, this creates a virtual env called `venv` which can be activated in Linux/Mac using `. ./venv/bin/activate`, if you're on Windows it's slightly different but you can find more information about virtual environments here https://realpython.com/python-virtual-environments-a-primer/#how-can-you-work-with-a-python-virtual-environment.

Once inside the virtual environment install all the necessary requirements with `pip install -r requirements.txt`. After this step you should be able to run `python main.py` and the app should run up.

### Caching Maps

- In `settings.py` set `LOAD_CACHED_MAP = False`
- Run the application once
- In `settings.py` set `LOAD_CACHED_MAP = True`
- Pypboy will now load the cached map on starting

### Enable app to startup on boot

```
pi@XXXX:~/Downloads/pypboy $ cat ~/launch_pipboy.sh
#!/bin/bash
cd ~/Downloads/pypboy
python ./main.py

pi@XXXX:~/Downloads/pypboy $ grep launch_pipboy /etc/lightdm/lightdm.conf
session-setup-script=/home/pi/launch_pipboy.sh
pi@XXXX:~/Downloads/pypboy $
```

## Authors

- Major overhaul by ZapWizard for the Functional Pip-Boy 3000 MK IV GUI

- Fixes and Updates by kingpinzs

- Fixes and Updates by amolloy

- Fixes and Updates by Goldstein

- Updates by Sabas of The Inventor's House Hackerspace

- Originally by grieve work original<br>

## License

MIT
