<template id="predictRetweet">
  <div class="addTweetContainer1">
    <textarea
      rows="1"
      cols="90"
      placeholder="TweetID"
      v-model="tweetID"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Username"
      v-model="username"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Timestamp"
      v-model="timestamp"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="#Followers"
      v-model="followers"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Friends"
      v-model="friends"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Favorites"
      v-model="favorites"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Entities"
      v-model="entities"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Positive Sentiment"
      v-model="POSsentiment"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Negative Sentiment"
      v-model="NEGsentiment"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Mentions"
      v-model="mentions"
    ></textarea>
    <textarea
      rows="1"
      cols="90"
      placeholder="Hashtags"
      v-model="hashtags"
    ></textarea>
    <div class="tweetLinkContainer">
      <a v-on:click="predictTweet()" class="tweetLink">Predict Retweet</a>
    </div>
    <p>PREDICTED NUMBER OF RETWEETS: {{ res }}</p>
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
    predictTweet: async function() {
      console.log(this.tweetID);
      console.log(this.username);
      console.log(this.timestamp);
      console.log(this.followers);
      console.log(this.friends);
      console.log(this.favorites);
      console.log(this.entities);
      console.log(this.POSsentiment);
      console.log(this.NEGsentiment);
      console.log(this.mentions);
      console.log(this.hashtags);
      await axios
        .post("http://localhost:5000/predict", {
          tweet: this.tweetID,
          username: this.username,
          timestamp: this.timestamp,
          followers: this.followers,
          friends: this.friends,
          favorites: this.favorites,
          entities: this.entities,
          POSsentiment: this.POSsentiment,
          NEGsentiment: this.NEGsentiment,
          mentions: this.mentions,
          hashtags: this.hashtags,
        })
        .then((response) => {
          console.log(
            "RESPONSE tweet FROM FLASK IS " +
              response.data.response_tweetID +
              " " +
              response.data.response_username
          );
          this.res = response.data;
        });
    },
  },
};
</script>

<style>
.addTweetContainer1 {
  max-width: 65%;
  text-align: center;
  display: flex;
  flex-direction: column;
}
.addTweetContainer1 textarea {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 15px;
  outline: 0px;
  padding: 10px 20px;
}
.addTweetContainer1 .tweetLink {
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
