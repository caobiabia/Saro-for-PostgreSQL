select  count(*) from comments as c,          badges as b,         users as u where u.Id = c.UserId 	and c.UserId = b.UserId  AND b.Date<='2014-09-10 22:50:06'::timestamp  AND c.CreationDate>='2010-07-27 14:02:33'::timestamp  AND c.CreationDate<='2014-09-11 00:54:08'::timestamp  AND u.Reputation>=1  AND u.Reputation<=234  AND u.UpVotes>=0  AND u.UpVotes<=368;