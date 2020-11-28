<template>
  <div>
      <form class="form-signin" v-on:submit.prevent="onSubmit">
        <h1>Drop image url below:</h1>
        <div class="form-group">
          <input type="text" placeholder="Image URL" class="form-control" v-model="url" />
        </div>
        <div class="form-group" v-if="!results && !waiting">
          <input type="submit" class="btn btn-primary" value="submit">
        </div>
        <div v-if="waiting">
          <i class="fa fa-circle-o-notch fa-spin fa-3x fa-fw"></i>
        </div>
        <div v-if="results">
          {{results}}
        </div>
        <br />
        <img :src="url" v-if="url" style="max-width: 100%;" />
      </form>
    </div>
</template>

<script>
    import superagent from 'superagent'
    
    var apiBaseUrl = 'http://localhost:7071'

    export default {
        data: function() {
            return {
                url: '',
                results: null,
                waiting: false
            }
        },
        methods: {
            onSubmit: function () {
            this.waiting = true;
            superagent
                .get(apiBaseUrl + '/api/classify')
                .query({ img: this.url })
                .end(function (err, res) {
                this.waiting = false;
                if (err) {
                    this.results = null;
                    alert("An error has occurred");
                } else {
                    this.results = res.body;
                }
                }.bind(this));
            }
        },
        watch: {
            url: function () {
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
</style>