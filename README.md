# Phishing Email Detector

start the python3 server
```$> python3 ./py/main.py
[!] Training models
.....
[+] Done
```
start the node.js UI server

```$> node ./js/index.js```

access the web UI on `localhost:3000`

for bert classifier, run the bert.py separately to ensure it does not freeze the server
```
$> python3 ./py/bert/bert.py
```

once the model has been saved, it can be loaded on the python3 server
