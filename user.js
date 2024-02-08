const express = require('express');
const session = require('express-session');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

const app = express();
const PORT = 3000;

// Connect to MongoDB (replace 'mongodb://localhost:27017/mydatabase' with your MongoDB connection string)
mongoose.connect('mongodb+srv://itsmerags:Mongodb1@login.et0smbr.mongodb.net/', { useNewUrlParser: true, useUnifiedTopology: true });

// Define a user schema
const userSchema = new mongoose.Schema({
  email: String,
  password: String,
});

const User = mongoose.model('User', userSchema);

app.use(express.urlencoded({ extended: true }));
app.use(session({ secret: 'your-secret-key', resave: true, saveUninitialized: true }));

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/login.html');
});

app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });

    if (user && bcrypt.compareSync(password, user.password)) {
      req.session.email = email;
      res.redirect('/welcome');
    } else {
      res.send('Invalid email or password');
    }
  } catch (error) {
    res.status(500).send('Internal Server Error');
  }
});

app.post('/register', async (req, res) => {
  const { email, password } = req.body;

  try {
    const existingUser = await User.findOne({ email });

    if (existingUser) {
      res.send('Email already registered');
    } else {
      const hashedPassword = bcrypt.hashSync(password, 10);
      const newUser = new User({ email, password: hashedPassword });
      await newUser.save();
      res.redirect('/');
    }
  } catch (error) {
    res.status(500).send('Internal Server Error');
  }
});

app.get('/welcome', (req, res) => {
  if (req.session.email) {
    res.send(`Welcome, ${req.session.email}!`);
  } else {
    res.redirect('/');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
