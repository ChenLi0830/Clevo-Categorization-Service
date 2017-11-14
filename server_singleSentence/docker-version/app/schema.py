import graphene
from api import categorization
import sys


class Query(graphene.ObjectType):
    categorizeSentence = graphene.String(
        text=graphene.String(required=True),
        category_list=graphene.Argument(
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
    )

    def resolve_categorizeSentence(self, info, text, category_list):
        print('category_list', category_list, file=sys.stderr)
        return categorization(text, category_list)


schema = graphene.Schema(query=Query)
