from django.db import models
from django.utils.translation import ugettext_lazy as _

import django
django.setup()

class Comments(models.Model):
    description = models.TextField(verbose_name=_('description'))
    positive = models.BooleanField(null=True, verbose_name=_('positive?'))
    
    def __str__(self):
        return self.description

    class Meta:
        db_table = 'comments'
        verbose_name = _('comments')
        verbose_name_plural = _('comments')