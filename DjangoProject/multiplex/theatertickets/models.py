from django.db import models

from multiplex.m_users.models import MUser
from multiplex.showtimes.models import Showtime
from multiplex.theaters.models import Theater


class TheaterTicket(models.Model):
    use_in_migrations = True
    theater_ticket_id =models.AutoField(primary_key=True)
    x = models.IntegerField()
    y = models.IntegerField()

    theater = models.ForeignKey(Theater, on_delete=models.CASCADE)
    showtime = models.ForeignKey(Showtime, on_delete=models.CASCADE)
    multiplex_user = models.ForeignKey(MUser, on_delete=models.CASCADE)

    class Meta:
        db_table = "multiplex_theater_tickets"

    def __str__(self):
        return f'{self.pk}{self.x}{self.y}'
