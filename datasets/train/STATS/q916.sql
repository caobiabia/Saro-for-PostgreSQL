select  count(*) from votes as v,          badges as b,         users as u where u.Id = v.UserId 	and v.UserId = b.UserId  AND b.Date>='2010-09-08 05:17:20'::timestamp  AND b.Date<='2014-09-09 16:35:52'::timestamp  AND u.Views>=0  AND u.Views<=53  AND v.VoteTypeId=2  AND v.CreationDate<='2014-09-12 00:00:00'::timestamp;