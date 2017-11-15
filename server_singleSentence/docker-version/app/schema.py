import graphene
from api import categorization
import sys


class Query(graphene.ObjectType):
    placeholder = graphene.String()

    def resolve_placeholder(self, info):
        return "This is a placeholder query, use the mutation instead"


class categoryResult(graphene.ObjectType):
    categoriesOfSentence = graphene.List(graphene.String)


class CategorizeSentence(graphene.Mutation):
    class Arguments:
        # geo = GeoInput(required=True)
        text = graphene.String(required=True)
        category_list = graphene.Argument(
            graphene.List(graphene.String),
            default_value=['结束询问',
                           '开头语',
                           '扣款话术3联系商家',
                           '收集信息',
                           '等待致谢',
                           '扣款话术1亲友儿童',
                           '问题回答',
                           '等待提示',
                           '扣款话术2号码保护']
        )

    Output = categoryResult

    def mutate(self, info, text, category_list):
        # print('category_list', category_list, file=sys.stderr)
        result = [categorization(text, category_list)]
        print('categorizationResult', result, file=sys.stderr)
        return categoryResult(categoriesOfSentence=result)


class Mutation(graphene.ObjectType):
    categorizeSentence = CategorizeSentence.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
