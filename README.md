# CATCH Analysis Tools
Data analysis tools for CATCH, an astronomical survey search tool


## Testing

Get required packages useful for testing:
```
pip install .[tests]
```

Running the tests:
```
tox -e py312-test
```

Running the tests with remote tests enabled, here for python 3.12:
```
tox -e py312-test -- --remote-data
```

Replacing 312 with something appropriate for your system.  A list of all testing environments can be viewed with
```
tox -l
```
