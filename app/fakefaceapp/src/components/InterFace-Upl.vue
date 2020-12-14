<template>
    <div>
        <form class="form-signin" v-on:submit.prevent="onSubmit">
          <h1>Upload your file here:</h1>
          <div class="form-group">
            <input type="file" name="image" @change="onFileSelected"/>
          </div>
          <div class="form-group" v-if="!results && !waiting">
            <input type="submit" class="btn btn-primary" value="submit">
          </div>
          <div v-if="waiting">
            <i class="fa fa-circle-o-notch fa-spin fa-3x fa-fw"></i>
          </div>
          <img :src="url2" v-if="url2" style="max-width: 100%;" />
          <div v-if="results" :style="resStyle">
            <h1>{{results}}</h1>
          </div>
          <br />
        </form>
      </div>
  </template>
  
  <script>
      import superagent from 'superagent'
      // TODO: Updated apiBaseUrl
      // var apiBaseUrl = 'http://localhost:7071'
      var apiBaseUrl = 'https://ffdetector.azurewebsites.net/'
  
      export default {
          data: function() {
              return {
                  url2: '',
                  results: null,
                  waiting: false,
                  resStyle: 'color: black;',
                  selectedFile: null,
              }
          },
          methods: {
              onSubmit() {
                this.waiting = true;
                var fd = new FormData();
                fd.append('img', this.selectedFile)

                superagent
                    .post(apiBaseUrl + 'api/classify')
                    .accept('application/json')
                    .send(fd)
                    .end(function (err, res) {
                        this.waiting = false;
                        if (err) {
                            this.results = null;
                            alert("An error has occurred");
                        } else {
                        res.body["result"] == "Not a fake" ? this.resStyle = 'color: green' : this.resStyle = 'color: red';
                        this.results = res.body["result"];
                        }
                    }.bind(this));
              },
              onFileSelected(event) {
                this.selectedFile = event.target.files[0]
                this.results = null;
                this.url2 = URL.createObjectURL(this.selectedFile);
              },
          },
          watch: {
              url2: function () {
              this.results = null;
              }
          }
      }
  </script>
  
  <style scoped>
      [v-cloak] {
        display: none;
      }
  
      i.fa-check {
        color: #009900;
      }
  
      i.fa-times {
        color: #aa0000;
      }
  
      @-ms-viewport {
        width: device-width;
      }
  
      @-o-viewport {
        width: device-width;
      }
  
      @viewport {
        width: device-width;
      }
  
      body {
        padding-top: 40px;
        padding-bottom: 40px;
      }
  
      #app {
        max-width: 475px;
      }
  
      .form-signin,
      #success {
        max-width: 430px;
        padding: 15px;
        margin: 0 auto;
      }
  
      .form-signin .form-signin-heading,
      .form-signin .checkbox {
        margin-bottom: 10px;
      }
  
      .form-signin .checkbox {
        font-weight: normal;
      }
  
      .form-signin .form-control {
        position: relative;
        height: auto;
        -webkit-box-sizing: border-box;
        -moz-box-sizing: border-box;
        box-sizing: border-box;
        padding: 10px;
        font-size: 16px;
      }
  
      .form-signin .form-control:focus {
        z-index: 2;
      }
  
      a {
        cursor: pointer;
      }
  
      table {
        margin-top: 24px;
      }
  
      .resultNeg {
        color: green;
      }
  </style>