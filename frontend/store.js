const store = {
    data: {
        search: '',
        feed: [
            {
                body: 'Be less curious about people and more curious about ideas.',
                date: '1m',
                author: 'Marie Curie',
                avatar: './images/Marie_Curie.jpg'
            },
            {
                body: "I have not failed. I've just found 10,000 ways that won't work.",
                date: '2m',
                author: 'Thomas Edison',
                avatar: './images/Thomas_Edison.jpg'
            },
            {
                body: 'Life is as tedious as a twice-told tale, Vexing the dull ear of a drowsy man.',
                date: '6m',
                author: 'William Shakespeare',
                avatar: './images/Shakespeare.jpg'
            },
            {
                body: "When I do good, I feel good. When I do bad, I feel bad. That's my religion",
                date: '11m',
                author: 'Abraham Lincoln',
                avatar: './images/Abraham_Lincoln.jpg'
            },
            {
                body: 'Truth is ever to be found in simplicity, and not in the multiplicity and confusion of things.',
                date: '21m',
                author: 'Isaac Newton',
                avatar: './images/Isaac_Newton.jpg'
            },
            {
                body: 'I can calculate the motion of heavenly bodies, but not the madness of people.',
                date: '21m',
                author: 'Isaac Newton',
                avatar: './images/Isaac_Newton.jpg'
            },
            {
                body: 'I am a slow walker, but I never walk back.',
                date: '11m',
                author: 'Abraham Lincoln',
                avatar: './images/Abraham_Lincoln.jpg'
            },
            {
                body: 'I never see what has been done; I only see what remains to be done.',
                date: '15m',
                author: 'Marie Curie',
                avatar: './images/Marie_Curie.jpg'
            },
            {
                body: 'Hell is empty and all the devils are here.',
                date: '24m',
                author: 'William Shakespeare',
                avatar: './images/Shakespeare.jpg'
            },
            {
                body: 'Genius is one percent inspiration and ninety-nine percent perspiration.',
                date: '40m',
                author: 'Thomas Edison',
                avatar: './images/Thomas_Edison.jpg'
            }
        ]
    },
    addtweetAction (tweet) {
        this.data.feed.unshift(tweet);
    }
}
