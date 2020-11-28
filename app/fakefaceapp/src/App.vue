<template>
  <div id="app">
    <InterFace />
  </div>
</template>

<script>

import InterFace from './components/InterFace-API.vue'

export default {
  name: 'App',
  components: {
    InterFace
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
