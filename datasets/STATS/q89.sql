select  count(*) from comments as c,          postHistory as ph,          users as u where u.Id = c.UserId 	and c.UserId = ph.UserId  AND c.Score=0  AND c.CreationDate<='2014-09-09 21:14:09'::timestamp  AND u.DownVotes>=0  AND u.UpVotes>=0  AND u.UpVotes<=244;