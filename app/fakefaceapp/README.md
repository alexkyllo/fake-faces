# fakefaceapp

## Description
Designed to work with an API request.

## Libraries
- Vue w/@vue/cli
- Superagent

## Components
### InterFace
Appends the url dropped into the form field as a query to the API listed in the ```apiBaseUrl``` variable + "```/api/classify```". It then returns the results.

## Testing
While using localhost, drop a test image into the "public" folder. It can then be passed to the URL form field as "http://localhost:8080/testimg.jpg"
