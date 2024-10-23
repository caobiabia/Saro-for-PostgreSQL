select  count(*) from comments as c,          badges as b,         users as u where u.Id = c.UserId 	and c.UserId = b.UserId  AND u.DownVotes>=0  AND u.DownVotes<=3;
