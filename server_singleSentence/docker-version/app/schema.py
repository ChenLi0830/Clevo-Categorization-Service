import graphene
from api import categorization


class Query(graphene.ObjectType):
    categorizeSentence = graphene.String(text=graphene.String())

    def resolve_categorizeSentence(self, info, text):
        return categorization(text)


schema = graphene.Schema(query=Query)
