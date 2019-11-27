""" This modules contains the services to handle the models.Comments """

import numpy as np
from django.db.models import Case, CharField, Value, When
from .comments_model import Comments

class CommentsService:

    @staticmethod
    def get_labeled_comments():
        """
        Get all labeled news
        :return: a list with 2 lists one with comments texts and another with labels
        """
        comments = Comments.objects.exclude(positive__isnull=True) \
                          .annotate(
                               label=Case(
                                   When(positive=True, then=Value('positive')),
                                   When(positive=False, then=Value('negative')),
                                   output_field=CharField(),
                               )
                           ) \
                          .values_list('description', 'label')

        return np.array(comments).T.tolist()
