select  count(*) from comments as c,          votes as v,  		badges as b,  		users as u where u.Id  = c.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND c.Score=1  AND c.CreationDate>='2010-09-30 08:43:11'::timestamp  AND u.DownVotes<=0  AND v.CreationDate<='2014-09-10 00:00:00'::timestamp;