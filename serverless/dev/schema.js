// import 'babel-polyfill'; // needed to for async/await
const graphql = require('graphql');
const {
    GraphQLSchema,
    GraphQLObjectType,
    GraphQLString,
    GraphQLID,
    GraphQLInt
} = graphql;

const _ = require("lodash");

const users = [
  {id: "123", firstName: "Chen", age: "29"},
  {id: "122", firstName: "May", age: "10"},
];

// import QueryType from './types/query';
// import MutationType from './types/mutation';


const UserType = new GraphQLObjectType({
  name: "User",
  fields: {
    id: {type: GraphQLID},
    firstName: {type: GraphQLString},
    age: {type: GraphQLInt}
  }
});

const rootQuery = new GraphQLObjectType({
  name: 'RootQueryType',
  fields: {
    user: {
      type: UserType,
      args: {
        id: {type: GraphQLString},
      },
      resolve: (parentValue, args) => {
        return _.find(users, {id: args.id});
      }
    }
  }
});


module.exports = new GraphQLSchema({
  query: rootQuery
  // query: QueryType,
  // mutation: MutationType,
});
