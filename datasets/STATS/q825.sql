select  count(*) from comments as c,          votes as v,  		badges as b,  		users as u where u.Id  = c.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND c.Score=0  AND c.CreationDate<='2014-09-09 20:49:28'::timestamp  AND u.Reputation<=441  AND u.DownVotes=0  AND u.UpVotes=2  AND v.VoteTypeId=5  AND v.CreationDate>='2010-07-19 00:00:00'::timestamp;