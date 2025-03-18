Will update this soon with more instructions & updated file paths. feel free to poke around with what's here rn:

project_root/
│
├── static/
│   └── plot.html  # plot placeholder that gets updated at the press of a button
│   └── styles.css # css styling code
│   └── description.html      # description that gets displayed on the fancy interface. Feel free to edit anything here to add context :)
│
├── templates/
│   └── index.html
│
├── app.py          # Flask app backend
├── main.py         # Entropy script (outputs static/plot.html)
├── kernel.py       # Kernel density script (outputs static/plot.html)
├── (your .py file here)       # Add anything else you wish to test
└── app.ipynb       # Jupyter Notebook that can run any .py file and display user params (filename, .hdf5 file)
