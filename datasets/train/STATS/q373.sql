select  count(*) from comments as c,          badges as b,         users as u where u.Id = c.UserId 	and c.UserId = b.UserId  AND c.Score=0  AND u.Reputation>=1  AND u.Reputation<=133;
