select  count(*) from comments as c,          votes as v,  		badges as b,  		users as u where u.Id  = c.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND c.Score=1  AND c.CreationDate>='2010-07-25 11:24:42'::timestamp  AND c.CreationDate<='2014-09-13 02:47:26'::timestamp  AND u.DownVotes>=0  AND u.UpVotes>=0  AND u.CreationDate>='2010-11-08 11:29:14'::timestamp;