select  count(*) from comments as c,  		posts as p,  		postHistory as ph,          votes as v,          users as u  where u.Id = c.UserId 	and c.UserId = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = v.UserId  AND p.PostTypeId=1  AND p.CommentCount<=11  AND p.CreationDate='2014-02-10 17:01:10'::timestamp  AND u.Views>=0  AND u.Views<=49  AND u.DownVotes<=0  AND u.CreationDate>='2010-12-03 16:15:57'::timestamp;