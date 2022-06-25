# Build docker image locally and run in kubernetes

## Building docker image in minikube

### Start minikube
* minikube start

#

### Set docker env
* eval $(minikube docker-env)             # unix shells
* minikube docker-env | Invoke-Expression # PowerShell

#

### Build image
* docker build -t image_name .

#

## Running Kubernetes cluster

### Deployment
* Blueprint of a deployment
* specify imagePullPolicy: Never in spec to use local docker image
* Example deployment configuration is in soil-app.yaml file

#

### Service
* To add deployments and service in same file add --- and press enter 
* Then start writing the service configurations
* Service is used to communicate with the pods running the application
* There are no deployments without service
* Add type: LoadBalancer  in spec and assign nodePort inside port in spec to create external service
* nodePort has range 30000 - 32767
* Note:- In minikube we have to set external ip address for external service using 
    ```
    minikube service name_of_service
    ```

#

###
```Use command kubectl apply -f file_name.yaml to create the object [deployment, secret, service , etc]```

#

#
```use pip install --no-cache-dir -r requirements.txt in docker file to avoid caching python packages and reduce image size```
#
