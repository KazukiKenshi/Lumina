const mongoose = require('mongoose');
require('dotenv').config();

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/lumina';

async function dropUsernameIndex() {
  try {
    await mongoose.connect(MONGODB_URI);
    console.log('Connected to MongoDB');

    const collection = mongoose.connection.db.collection('users');
    
    try {
      await collection.dropIndex('username_1');
      console.log('✅ Username index dropped successfully');
    } catch (err) {
      if (err.code === 27) {
        console.log('✅ Username index does not exist (already removed)');
      } else {
        console.error('❌ Error dropping index:', err.message);
      }
    }

    await mongoose.connection.close();
    console.log('Disconnected from MongoDB');
    process.exit(0);
  } catch (err) {
    console.error('❌ Connection error:', err.message);
    process.exit(1);
  }
}

dropUsernameIndex();
