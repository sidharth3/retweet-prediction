<template id="addTweet">
  <div class="addTweetContainer">
    <textarea
      rows="7"
      cols="90"
      placeholder="What’s happening?"
      v-model="tweet"
    >
    </textarea>

    <div class="tweetLinkContainer">
      <a v-on:click="postTweet()" class="tweetLink">Tweet</a>
    </div>
    <p>{{ res }}</p>
  </div>
</template>

<script>
module.exports = {
  data: function() {
    return {
      tweet: "",
      res: "",
    };
  },
  methods: {
    postTweet: async function() {
      // await axios
      //   .post("http://localhost:5000/predict", { tweet: this.tweet })
      //   .then((response) => {
      //     console.log(response.data);
      //     // need to convert response data to string
      //     if (response.data == "NOT OFFENSIVE") {
      //       const tweet = {
      //         body: this.tweet,
      //         date: "now",
      //         author: "Me",
      //         avatar: "./images/egg.jpg",
      //       };
      //       this.res = response.data;
      //       store.addtweetAction(tweet);
      //     } else {
      //       // offensive
      //       this.res = response.data;
      //       alert("OFFENSIVE LANGUAGE DETECTED");
      //     }
      //   });
      const tweet = {
        body: this.tweet,
        date: "now",
        author: "Me",
        avatar: "./images/egg.jpg",
      };
      store.addtweetAction(tweet);
    },
  },
};
</script>

<style>
.addTweetContainer {
  max-width: 65%;
  text-align: center;
  display: flex;
  flex-direction: column;
}
.addTweetContainer textarea {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 15px;
  outline: 0px;
  padding: 10px 20px;
}
.addTweetContainer .tweetLink {
  background-color: #1da1f2;
  border: 1px solid #1da1f2;
  color: #fff;
  text-decoration: none;
  border-radius: 100px;
  padding: 6px 16px;
  margin-left: 15px;
  width: 50px;
}
.tweetLinkContainer {
  text-align: right;
  margin-top: 30px;
}
</style>
