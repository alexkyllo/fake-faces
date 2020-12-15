<template>
  <div id="app">
    <img class="cover" src="cover.png" v-if="!selectedOpt"/>
    <InterFaceFile v-if="selectedOpt=='Upload'"/>
    <InterFaceAPI v-if="selectedOpt=='URL'"/>
    <div id="radiobox">
      <input type = "radio" name = "rad" id="Upload" value = "Upload" v-model="selectedOpt"><label for="Upload">Upload</label><br>
      <input type = "radio" name = "rad" id="Upload" value = "URL" v-model="selectedOpt"><label for="URL">URL</label>
    </div>
  </div>
</template>

<script>

import InterFaceFile from './components/InterFace-Upl.vue'
import InterFaceAPI from './components/InterFace-API.vue'

export default {
  name: 'App',
  data: function() {
    return {
      selectedOpt: null,
    }
  },
  components: {
    InterFaceAPI,
    InterFaceFile
  },

}
</script>

<style>
#app {
  
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
  width: 40%;
  margin: auto;
}
#radiobox {
  text-align: left;
}

#Upload {
  margin: 5px;
}

.cover {
  padding: 30px 0px 30px 0px;
  width: 100%;
}
</style>
