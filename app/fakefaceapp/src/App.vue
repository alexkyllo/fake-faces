<template>
  <div id="app">
    <InterFaceFile />
    <InterFaceAPI />
  </div>
</template>

<script>

import InterFaceFile from './components/InterFace-Upl.vue'
import InterFaceAPI from './components/InterFace-API.vue'

export default {
  name: 'App',
  components: {
    InterFaceAPI,
    InterFaceFile
  }
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
}
</style>
