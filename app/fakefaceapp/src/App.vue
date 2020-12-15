<template>
  <div id="app">
    <img class="cover" src="cover.png" v-if="!selectedOpt"/>
    <InterFaceFile v-if="selectedOpt=='Upload'"/>
    <InterFaceAPI v-if="selectedOpt=='URL'"/>
    <div id="radiobox">
      <div>
        <input type = "radio" name = "rad" id="Upload" value = "Upload" v-model="selectedOpt"><label for="Upload">Upload</label>
      </div>
      <div>
        <input type = "radio" name = "rad" id="Upload" value = "URL" v-model="selectedOpt"><label for="URL">URL</label>
      </div>
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
  display: flex;
  align-items: center;
  justify-content: center;
  flex-flow: column;
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;

}
#radiobox {
  text-align: left;
  width: 500px;
}

#Upload {
  margin: 5px;
}

.cover {
  padding: 30px 0px 30px 0px;
  max-width: 500px;
  margin: auto;
}
</style>
