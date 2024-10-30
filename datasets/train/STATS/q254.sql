select  count(*) from comments as c,          postHistory as ph,          users as u where u.Id = c.UserId 	and c.UserId = ph.UserId  AND u.Reputation>=1  AND u.Reputation<=169;
