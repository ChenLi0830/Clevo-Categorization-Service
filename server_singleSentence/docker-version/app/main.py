#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8
from flask import Flask
from api import *
import os

from flask_graphql import GraphQLView
from schema import schema

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app.add_url_rule('/graphql', view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True))

if __name__ == "__main__":
    # decide what port to run the app in
    # port = int(os.environ.get('PORT', 7271))
    # app.run(host='127.0.0.1', port=port)
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=80)
    # optional if we want to run in debugging mode
    # app.run(debug=True)
