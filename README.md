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


## Hosting

The CAT may be run as a containerized service, hosted locally or by AWS Fargate.

### Local

Running locally will install the currently checked out version of the CAT.

1. Copy env-template to .env and edit.
    a. Set TF_VAR_CAT_DEPLOYMENT to "local"
2. bash _build_container
3. bash _test_local
