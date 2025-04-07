app/
│
├── static/
│   └── plot.html
│   └── styles.css
│   └── description.html
│   └── segmentation.html
│
├── templates/
│   └── index.html
│
├── algos/
│   └── entropy.py  # Entropy script (outputs static/plot.html)
│   └── kernel.py   # Kernel density script (outputs static/plot.html)
│   └── kernel.py   # segmentation script (outputs static/plot.html)
│
├── templates/
│   └── play_pushing.hdf5             # play demos
│   └── expert_lampshade2_demos.hdf5  # expert demos
│
├── app.py                 # Flask app backend
└── requirements.txt       # All possible dependencies for the programs in this app to be run *Needs update

Instructions - run app.py (python3 app.py) then ctrl+click on the server running in the terminal/ type it into a web url. It should open in your web browser. From there, type in the file path to hdf5 file. (data/play_pushing.hdf5 for example) and click a button to graph.
