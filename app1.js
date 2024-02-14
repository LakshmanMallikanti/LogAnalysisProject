const express = require('express');
const session = require('express-session');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const path = require('path');

require('dotenv').config()

const app = express();
const PORT = process.env.PORT || 3000;

mongoose.connect(process.env.MONGODB_URI);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use( express.static(path.join('PS')));
app.use( express.static(path.join(__dirname, '/login.css')));
app.use( express.static(path.join(__dirname, '/login.js')));
app.use( express.static(path.join(__dirname, '/webstyle.css')));
app.use(session({
    secret: process.env.SESSION_SECRET, // Replace this with your actual secret key
    resave: false,
    saveUninitialized: true,
  }));
  
const User = mongoose.model('User', new mongoose.Schema({
    username:String,
  email: String,
  password: String,
}));

app.use(session({
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: true,
}));

app.get('/', (req, res) => {
  res.sendFile('login.html', { root: path.join(__dirname) });
});

app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });

    if (user && await bcrypt.compare(password, user.password)) {
      req.session.email = email;
      res.redirect('/welcome');
    } else {
      res.send('Invalid email or password');
    }
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  }
});

app.post('/register', async (req, res) => {
    const { username, email, password } = req.body;
  
    try {
      const existingUser = await User.findOne({ email });
  
      if (existingUser) {
        res.send('Email already registered');
      } else {
        if (password === '') {
          return res.status(400).json({ message: 'Password is required' });
        }
  
        console.log('Username:', username);
        console.log('Email:', email);
        console.log('Password:', password);
  
        const hashedPassword = await bcrypt.hash(password, 10);
        console.log('Hashed Password:', hashedPassword);
  
        const newUser = new User({ username, email, password: hashedPassword });
        await newUser.save();
  
        res.redirect('/web.html');
      }
    } catch (error) {
      console.error(error);
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

app.get('/web.html', (req, res) => {
  res.sendFile('web.html', { root: path.join(__dirname, 'public') });
});


//app.use('/webScript.js', express.static(path.join(__dirname, '/public/webScript.js')));

// Add profile route
app.get('/profile', (req, res) => {
  // Assuming you have user data available
  const userProfileData = {
    email: req.session.email || 'Guest User',
    // Add other user data as needed
  };

  res.sendFile('profile.html', { root: path.join(__dirname, 'public') });
});
// ... (existing code) ...

// Endpoint to get user information
app.get('/get-user-info', (req, res) => {
  if (req.session.email) {
      // If the user is logged in, send their information
      const user = { email: req.session.email, photo: req.session.photo };
      res.json(user);
  } else {
      // If the user is not logged in, send an empty response
      res.json({});
  }
});

// Endpoint to handle changing password
app.post('/change-password', async (req, res) => {
  try {
      if (req.session.email) {
          // Implement your logic to change the password
          // For example, you can use bcrypt to hash the new password and update it in the database
          const newPassword = req.body.newPassword;
          const hashedPassword = await bcrypt.hash(newPassword, 10);

          // Update the user's password in the database
          await User.updateOne({ email: req.session.email }, { $set: { password: hashedPassword } });

          res.send('Password changed successfully');
      } else {
          // If the user is not logged in, send an error response
          res.status(401).send('Unauthorized');
      }
  } catch (error) {
      console.error(error);
      res.status(500).send('Internal Server Error');
  }
});

// Endpoint to handle logout
app.get('/logout', (req, res) => {
  req.session.destroy((err) => {
      if (err) {
          console.error(err);
          res.status(500).send('Internal Server Error');
      } else {
          res.redirect('/');
      }
  });
});

// ... (existing code) ...


app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
