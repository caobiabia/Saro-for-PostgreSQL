select  count(*) from comments as c,          badges as b,         users as u where u.Id = c.UserId 	and c.UserId = b.UserId  AND b.Date>='2010-07-19 22:49:06'::timestamp;
