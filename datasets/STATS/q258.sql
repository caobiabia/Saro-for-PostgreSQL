select  count(*) from comments as c,          votes as v,  		badges as b,  		users as u where u.Id  = c.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND c.Score=3  AND u.DownVotes>=0  AND u.UpVotes<=1408  AND u.CreationDate>='2010-12-18 11:53:45'::timestamp  AND v.VoteTypeId=2;