const webpack = require('webpack');
const path = require('path');

const BUILD_DIR = path.resolve(__dirname, 'graphiQLClient/dist');
const APP_DIR = path.resolve(__dirname, 'graphiQLClient');

var config = {
  entry: APP_DIR + '/GraphiQL.js',
  output: {
    path: BUILD_DIR,
    filename: 'bundle.js'
  },
  module: {
    loaders: [
      {
        test: /\.jsx?/,
        include: APP_DIR,
        loader: 'babel-loader',
        exclude: /node_modules/
      },
    ]
  },
  resolve: {
    extensions: ['.js', '.jsx'],
  },
  plugins: [
    // new webpack.DefinePlugin({
    //   'process.env': {
    //     'NODE_ENV': JSON.stringify('production')
    //   }
    // })
  ],
};

module.exports = config;